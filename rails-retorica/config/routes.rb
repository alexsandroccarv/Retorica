# -*- encoding : utf-8 -*-
Retorica::Application.routes.draw do

  root :to => 'application#index'

  resources :dashboards, :expect => [:new, :create, :edit, :delete]

  resources :deputados, :except => [:new, :create, :edit, :delete] do
    collection { get :import }
    collection { get :discursos }
  end

end
