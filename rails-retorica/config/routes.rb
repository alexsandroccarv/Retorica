# -*- encoding : utf-8 -*-
Retorica::Application.routes.draw do

  root :to => redirect('/dashboards/first')

  resources :dashboards, :except => [:new, :create, :edit, :delete] do
    collection { get :first }
  end

  resources :deputados, :except => [:new, :create, :edit, :delete]

end
